use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rustc_data_structures::sync::IntoDynSyncSend;
use rustc_fs_util::try_canonicalize;
use rustc_proc_macro::bridge::client::Client as ProcMacroClient;
use rustc_proc_macro::bridge::server::DynClient;
use rustc_session::StableCrateId;
use tracing::debug;

use crate::locator::CrateError;

fn format_dlopen_err(e: &(dyn std::error::Error + 'static)) -> String {
    e.sources().map(|e| format!(": {e}")).collect()
}

fn attempt_load_dylib(path: &Path) -> Result<libloading::Library, libloading::Error> {
    #[cfg(target_os = "aix")]
    if let Some(ext) = path.extension()
        && ext.eq("a")
    {
        // On AIX, we ship all libraries as .a big_af archive
        // the expected format is lib<name>.a(libname.so) for the actual
        // dynamic library
        let library_name = path.file_stem().expect("expect a library name");
        let mut archive_member = std::ffi::OsString::from("a(");
        archive_member.push(library_name);
        archive_member.push(".so)");
        let new_path = path.with_extension(archive_member);

        // On AIX, we need RTLD_MEMBER to dlopen an archived shared
        let flags = libc::RTLD_LAZY | libc::RTLD_LOCAL | libc::RTLD_MEMBER;
        return unsafe { libloading::os::unix::Library::open(Some(&new_path), flags) }
            .map(|lib| lib.into());
    }

    unsafe { libloading::Library::new(&path) }
}

// On Windows the compiler would sometimes intermittently fail to open the
// proc-macro DLL with `Error::LoadLibraryExW`. It is suspected that something in the
// system still holds a lock on the file, so we retry a few times before calling it
// an error.
fn load_dylib(path: &Path, max_attempts: usize) -> Result<libloading::Library, String> {
    assert!(max_attempts > 0);

    let mut last_error = None;

    for attempt in 0..max_attempts {
        debug!("Attempt to load proc-macro `{}`.", path.display());
        match attempt_load_dylib(path) {
            Ok(lib) => {
                if attempt > 0 {
                    debug!(
                        "Loaded proc-macro `{}` after {} attempts.",
                        path.display(),
                        attempt + 1
                    );
                }
                return Ok(lib);
            }
            Err(err) => {
                // Only try to recover from this specific error.
                if !matches!(err, libloading::Error::LoadLibraryExW { .. }) {
                    debug!("Failed to load proc-macro `{}`. Not retrying", path.display());
                    let err = format_dlopen_err(&err);
                    // We include the path of the dylib in the error ourselves, so
                    // if it's in the error, we strip it.
                    if let Some(err) = err.strip_prefix(&format!(": {}", path.display())) {
                        return Err(err.to_string());
                    }
                    return Err(err);
                }

                last_error = Some(err);
                std::thread::sleep(Duration::from_millis(100));
                debug!("Failed to load proc-macro `{}`. Retrying.", path.display());
            }
        }
    }

    debug!("Failed to load proc-macro `{}` even after {} attempts.", path.display(), max_attempts);

    let last_error = last_error.unwrap();
    let message = if let Some(src) = last_error.source() {
        format!("{} ({src}) (retried {max_attempts} times)", format_dlopen_err(&last_error))
    } else {
        format!("{} (retried {max_attempts} times)", format_dlopen_err(&last_error))
    };
    Err(message)
}

pub enum DylibError {
    DlOpen(String, String),
    DlSym(String, String),
}

impl From<DylibError> for CrateError {
    fn from(err: DylibError) -> CrateError {
        match err {
            DylibError::DlOpen(path, err) => CrateError::DlOpen(path, err),
            DylibError::DlSym(path, err) => CrateError::DlSym(path, err),
        }
    }
}

pub unsafe fn load_symbol_from_dylib<T: Copy>(
    path: &Path,
    sym_name: &str,
) -> Result<T, DylibError> {
    // Make sure the path contains a / or the linker will search for it.
    let path = try_canonicalize(path).unwrap();
    let lib =
        load_dylib(&path, 5).map_err(|err| DylibError::DlOpen(path.display().to_string(), err))?;

    let sym = unsafe { lib.get::<T>(sym_name.as_bytes()) }
        .map_err(|err| DylibError::DlSym(path.display().to_string(), format_dlopen_err(&err)))?;

    // Intentionally leak the dynamic library. We can't ever unload it
    // since the library can make things that will live arbitrarily long.
    let sym = unsafe { sym.into_raw() };
    std::mem::forget(lib);

    Ok(*sym)
}

pub(crate) fn dlsym_proc_macros(
    path: &Path,
    stable_crate_id: StableCrateId,
) -> Result<Vec<IntoDynSyncSend<DynClient>>, DylibError> {
    let sym_name = rustc_session::generate_proc_macro_decls_symbol(stable_crate_id);
    debug!("trying to dlsym proc_macros {} for symbol `{}`", path.display(), sym_name);

    unsafe {
        // FIXME(bjorn3) this depends on the unstable slice memory layout
        let result = load_symbol_from_dylib::<*const &[ProcMacroClient]>(path, &sym_name);
        match result {
            Ok(result) => {
                debug!("loaded dlsym proc_macros {} for symbol `{}`", path.display(), sym_name);
                Ok((*result)
                    .iter()
                    .map(|proc_macro| IntoDynSyncSend(proc_macro.into_dyn_client()))
                    .collect())
            }
            Err(_err) => {
                let engine = wasmi::Engine::default();
                let module = wasmi::Module::new(&engine, std::fs::read(path).unwrap()).unwrap();

                let mut store =
                    wasmi::Store::new(&engine, wasmi_wasi::WasiCtxBuilder::new().build());
                let mut linker = wasmi::Linker::new(&engine);
                linker
                    .func_wrap("env", "__rustc_proc_macro_dispatch", |_: u32, _: u32| -> () {
                        unreachable!()
                    })
                    .unwrap();
                wasmi_wasi::add_to_linker(&mut linker, |ctx| ctx).unwrap();
                let instance = linker.instantiate_and_start(&mut store, &module).unwrap();

                let memory = instance.get_export(&store, "memory").unwrap().into_memory().unwrap();

                let sym = instance
                    .get_export(&store, &sym_name)
                    .unwrap()
                    .into_global()
                    .unwrap()
                    .get(&store)
                    .i32()
                    .unwrap();

                let mut data = [0; 8];
                memory.read(&store, sym as usize, &mut data).unwrap();
                let ptr = u32::from_le_bytes(data[..4].try_into().unwrap());
                let len = u32::from_le_bytes(data[4..8].try_into().unwrap());

                Ok((0..len)
                    .map(|i| {
                        let mut data = [0; 4];
                        memory.read(&store, (ptr + 4 * i) as usize, &mut data).unwrap();
                        let func_ptr = u32::from_le_bytes(data);

                        IntoDynSyncSend(wasm_macro_client(engine.clone(), module.clone(), func_ptr))
                    })
                    .collect::<Vec<_>>())
            }
        }
    }
}

fn wasm_macro_client(engine: wasmi::Engine, module: wasmi::Module, func_ptr: u32) -> DynClient {
    DynClient {
        run: Arc::new(move |config| {
            struct Ctx<'a> {
                wasi_ctx: wasmi_wasi::WasiCtx,
                dispatch: rustc_proc_macro::bridge::Closure<'a>,
                client_refs: Option<ClientRefs>,
            }

            #[derive(Clone)]
            struct ClientRefs {
                memory: wasmi::Memory,
                buffer_replace: wasmi::TypedFunc<(u32, u32), ()>,
                buffer_ptr: wasmi::TypedFunc<(u32,), (u32,)>,
                buffer_len: wasmi::TypedFunc<(u32,), (u32,)>,
            }

            impl ClientRefs {
                fn read(
                    &self,
                    store: &mut impl wasmi::AsContextMut,
                    buffer: u32,
                ) -> Result<Vec<u8>, wasmi::Error> {
                    let (buffer_ptr,) = self.buffer_ptr.call(&mut *store, (buffer,))?;
                    let (buffer_len,) = self.buffer_len.call(&mut *store, (buffer,))?;
                    let mut data = vec![0; buffer_len as usize];
                    self.memory.read(store, buffer_ptr as usize, &mut data)?;
                    Ok(data)
                }
            }

            let mut store = wasmi::Store::new(
                &engine,
                Ctx {
                    wasi_ctx: wasmi_wasi::WasiCtxBuilder::new()
                        .inherit_stdout()
                        .inherit_stderr()
                        .build(),
                    dispatch: config.dispatch,
                    client_refs: None,
                },
            );
            let mut linker: wasmi::Linker<Ctx<'_>> = wasmi::Linker::new(&engine);
            linker
                .func_new(
                    "env",
                    "__rustc_proc_macro_dispatch",
                    wasmi::FuncType::new([wasmi::ValType::I32, wasmi::ValType::I32], []),
                    |mut caller, inputs, _outputs| {
                        let client_refs = caller.data().client_refs.clone().unwrap();
                        let input_buffer = inputs[0].i32().unwrap().cast_unsigned();
                        let output_buffer = inputs[1].i32().unwrap().cast_unsigned();

                        let input_buffer_data = client_refs.read(&mut caller, input_buffer)?;

                        let output_buffer_data =
                            caller.data_mut().dispatch.call(input_buffer_data.into());

                        client_refs.buffer_replace.call(
                            &mut caller,
                            (output_buffer, output_buffer_data.len().try_into().unwrap()),
                        )?;
                        let (output_buffer_ptr,) =
                            client_refs.buffer_ptr.call(&mut caller, (output_buffer,))?;
                        client_refs.memory.write(
                            &mut caller,
                            output_buffer_ptr as usize,
                            &output_buffer_data,
                        )?;

                        Ok(())
                    },
                )
                .unwrap();
            wasmi_wasi::add_to_linker(&mut linker, |ctx| &mut ctx.wasi_ctx).unwrap();
            let instance = linker.instantiate_and_start(&mut store, &module).unwrap();

            fn get_func<T: wasmi::WasmParams, U: wasmi::WasmResults>(
                instance: &wasmi::Instance,
                store: &impl wasmi::AsContext,
                name: &str,
            ) -> wasmi::TypedFunc<T, U> {
                instance.get_export(store, name).unwrap().into_func().unwrap().typed(store).unwrap()
            }

            let client_refs = ClientRefs {
                memory: instance.get_export(&store, "memory").unwrap().into_memory().unwrap(),
                buffer_replace: get_func(&instance, &store, "__rustc_proc_macro_buffer_replace"),
                buffer_ptr: get_func(&instance, &store, "__rustc_proc_macro_buffer_ptr"),
                buffer_len: get_func(&instance, &store, "__rustc_proc_macro_buffer_len"),
            };
            store.data_mut().client_refs = Some(client_refs.clone());

            let alloc_buffer: wasmi::TypedFunc<(u32,), (u32,)> =
                get_func(&instance, &store, "__rustc_proc_macro_alloc_buffer");

            let call_client: wasmi::TypedFunc<(u32, u32), (u32,)> =
                get_func(&instance, &store, "__rustc_proc_macro_call_client");

            let (input_buffer,) =
                alloc_buffer.call(&mut store, (config.input.len().try_into().unwrap(),)).unwrap();
            let input_buffer_ptr =
                client_refs.buffer_ptr.call(&mut store, (input_buffer,)).unwrap().0;
            instance
                .get_export(&store, "memory")
                .unwrap()
                .into_memory()
                .unwrap()
                .write(&mut store, input_buffer_ptr as usize, &config.input)
                .unwrap();

            let (output_buffer,) = call_client.call(&mut store, (input_buffer, func_ptr)).unwrap();

            let output_buffer_data = client_refs.read(&mut store, output_buffer).unwrap();

            output_buffer_data.into()
        }),
    }
}
