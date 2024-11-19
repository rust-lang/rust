use std::fs::{File, Metadata, OpenOptions};
use std::io;
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use rustc_abi::Size;

use crate::shims::windows::handle::{Handle, EvalContextExt as _, PseudoHandle};
use crate::*;
use crate::shims::files::{FileDescription, FileDescriptionRef, EvalContextExt as _};
use crate::shims::time::system_time_to_duration;

#[derive(Debug)]
pub struct FileHandle {
    pub(crate) file: File,
    pub(crate) writable: bool,
}

impl FileDescription for FileHandle {
    fn name(&self) -> &'static str {
        "file"
    }

    fn read<'tcx>(
        &self,
        _self_ref: &FileDescriptionRef,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        let mut bytes = vec![0; len];
        let result = (&mut &self.file).read(&mut bytes);
        match result {
            Ok(read_size) => ecx.return_read_success(ptr, &bytes, read_size, dest),
            Err(e) => ecx.set_last_error_and_return(e, dest),
        }
    }

    fn write<'tcx>(
        &self,
        _self_ref: &FileDescriptionRef,
        communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        let bytes = ecx.read_bytes_ptr_strip_provenance(ptr, Size::from_bytes(len))?;
        let result = (&mut &self.file).write(bytes);
        match result {
            Ok(write_size) => ecx.return_write_success(write_size, dest),
            Err(e) => ecx.set_last_error_and_return(e, dest),
        }
    }

    fn seek<'tcx>(
        &self,
        communicate_allowed: bool,
        offset: SeekFrom,
    ) -> InterpResult<'tcx, io::Result<u64>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        interp_ok((&mut &self.file).seek(offset))
    }

    fn close<'tcx>(
        self: Box<Self>,
        communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        assert!(communicate_allowed, "isolation should have prevented even opening a file");
        // We sync the file if it was opened in a mode different than read-only.
        if self.writable {
            // `File::sync_all` does the checks that are done when closing a file. We do this to
            // to handle possible errors correctly.
            let result = self.file.sync_all();
            // Now we actually close the file and return the result.
            drop(*self);
            interp_ok(result)
        } else {
            // We drop the file, this closes it but ignores any errors
            // produced when closing it. This is done because
            // `File::sync_all` cannot be done over files like
            // `/dev/urandom` which are read-only. Check
            // https://github.com/rust-lang/miri/issues/999#issuecomment-568920439
            // for a deeper discussion.
            drop(*self);
            interp_ok(Ok(()))
        }
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, io::Result<Metadata>> {
        interp_ok(self.file.metadata())
    }

    fn is_tty(&self, communicate_allowed: bool) -> bool {
        communicate_allowed && self.file.is_terminal()
    }
}

#[derive(Debug)]
pub struct DirHandle {
    pub(crate) path: PathBuf,
}

impl FileDescription for DirHandle {
    fn name(&self) -> &'static str {
        "directory"
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, io::Result<Metadata>> {
        interp_ok(self.path.metadata())
    }
}

#[derive(Debug)]
pub struct MetadataHandle {
    pub(crate) path: PathBuf,
}

impl FileDescription for MetadataHandle {
    fn name(&self) -> &'static str {
        "metadata-only"
    }

    fn metadata<'tcx>(&self) -> InterpResult<'tcx, io::Result<Metadata>> {
        interp_ok(self.path.metadata())
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
#[allow(non_snake_case)]
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn CreateFileW(
        &mut self,
        file_name: &OpTy<'tcx>,            // LPCWSTR
        desired_access: &OpTy<'tcx>,       // DWORD
        share_mode: &OpTy<'tcx>,           // DWORD
        security_attributes: &OpTy<'tcx>,  // LPSECURITY_ATTRIBUTES
        creation_disposition: &OpTy<'tcx>, // DWORD
        flags_and_attributes: &OpTy<'tcx>, // DWORD
        template_file: &OpTy<'tcx>,        // HANDLE
    ) -> InterpResult<'tcx, Handle> {
        // ^ Returns HANDLE
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "CreateFileW");
        this.check_no_isolation("`CreateFileW`")?;

        let file_name = String::from_utf16_lossy(&this.read_wide_str(this.read_pointer(file_name)?)?);
        let file_name = Path::new(&file_name);
        let desired_access = this.read_scalar(desired_access)?.to_u32()?;
        let share_mode = this.read_scalar(share_mode)?.to_u32()?;
        let security_attributes = this.read_pointer(security_attributes)?;
        let creation_disposition = this.read_scalar(creation_disposition)?.to_u32()?;
        let flags_and_attributes = this.read_scalar(flags_and_attributes)?.to_u32()?;
        let template_file = this.read_target_usize(template_file)?;

        let generic_read = this.eval_windows_u32("c", "GENERIC_READ");
        let generic_write = this.eval_windows_u32("c", "GENERIC_WRITE");

        if desired_access & !(generic_read | generic_write) != 0 {
            throw_unsup_format!("CreateFileW: Unsupported access mode: {desired_access}");
        }

        let file_share_delete = this.eval_windows_u32("c", "FILE_SHARE_DELETE");
        let file_share_read = this.eval_windows_u32("c", "FILE_SHARE_READ");
        let file_share_write = this.eval_windows_u32("c", "FILE_SHARE_WRITE");

        if share_mode & !(file_share_delete | file_share_read | file_share_write) != 0 || share_mode == 0 {
            throw_unsup_format!("CreateFileW: Unsupported share mode: {share_mode}");
        }
        if !this.ptr_is_null(security_attributes)? {
            throw_unsup_format!("CreateFileW: Security attributes are not supported");
        }

        let create_always = this.eval_windows_u32("c", "CREATE_ALWAYS");
        let create_new = this.eval_windows_u32("c", "CREATE_NEW");
        let open_always = this.eval_windows_u32("c", "OPEN_ALWAYS");
        let open_existing = this.eval_windows_u32("c", "OPEN_EXISTING");
        let truncate_existing = this.eval_windows_u32("c", "TRUNCATE_EXISTING");

        if ![create_always, create_new, open_always, open_existing, truncate_existing].contains(&creation_disposition) {
            throw_unsup_format!("CreateFileW: Unsupported creation disposition: {creation_disposition}");
        }

        let file_attribute_normal = this.eval_windows_u32("c", "FILE_ATTRIBUTE_NORMAL");
        // This must be passed to allow getting directory handles. If not passed, we error on trying
        // to open directories below
        let file_flag_backup_semantics = this.eval_windows_u32("c", "FILE_FLAG_BACKUP_SEMANTICS");
        let file_flag_open_reparse_point = this.eval_windows_u32("c", "FILE_FLAG_OPEN_REPARSE_POINT");

        let flags_and_attributes = match flags_and_attributes {
            0 => file_attribute_normal,
            _ => flags_and_attributes,
        };
        if !(file_attribute_normal | file_flag_backup_semantics | file_flag_open_reparse_point) & flags_and_attributes != 0 {
            throw_unsup_format!("CreateFileW: Unsupported flags_and_attributes: {flags_and_attributes}");
        }

        if flags_and_attributes & file_flag_open_reparse_point != 0 && creation_disposition == create_always {
            throw_machine_stop!(TerminationInfo::Abort("Invalid CreateFileW argument combination: FILE_FLAG_OPEN_REPARSE_POINT with CREATE_ALWAYS".to_string()));
        }

        if template_file != 0 {
            throw_unsup_format!("CreateFileW: Template files are not supported");
        }

        let desired_read = desired_access & generic_read != 0;
        let desired_write = desired_access & generic_write != 0;
        let exists = file_name.exists();
        let is_dir = file_name.is_dir();

        if flags_and_attributes == file_attribute_normal && is_dir {
            this.set_last_error(IoError::WindowsError("ERROR_ACCESS_DENIED"))?;
            return interp_ok(Handle::Invalid);
        }

        let mut options = OpenOptions::new();
        if desired_read {
            options.read(true);
        }
        if desired_write {
            options.write(true);
        }

        if creation_disposition == create_always {
            if file_name.exists() {
                this.set_last_error(IoError::WindowsError("ERROR_ALREADY_EXISTS"))?;
            }
            options.create(true);
            options.truncate(true);
        } else if creation_disposition == create_new {
            options.create_new(true);
            if !desired_write {
                options.append(true);
            }
        } else if creation_disposition == open_always {
            if file_name.exists() {
                this.set_last_error(IoError::WindowsError("ERROR_ALREADY_EXISTS"))?;
            }
            options.create(true);
        } else if creation_disposition == open_existing {
            // Nothing
        } else if creation_disposition == truncate_existing {
            options.truncate(true);
        }

        let handle = if is_dir && exists {
            let fh = &mut this.machine.fds;
            let fd = fh.insert_new(DirHandle { path: file_name.into() });
            Ok(Handle::File(fd as u32))
        } else if creation_disposition == open_existing && desired_access == 0 {
            // Windows supports handles with no permissions. These allow things such as reading
            // metadata, but not file content.
            let fh = &mut this.machine.fds;
            let fd = fh.insert_new(MetadataHandle { path: file_name.into() });
            Ok(Handle::File(fd as u32))
        } else {
            options.open(file_name).map(|file| {
                let fh = &mut this.machine.fds;
                let fd = fh.insert_new(FileHandle { file, writable: desired_write });
                Handle::File(fd as u32)
            })
        };

        match handle {
            Ok(handle) => interp_ok(handle),
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(Handle::Invalid)
            }
        }
    }

    fn GetFileInformationByHandle(
        &mut self,
        file: &OpTy<'tcx>,             // HANDLE
        file_information: &OpTy<'tcx>, // LPBY_HANDLE_FILE_INFORMATION
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns BOOL (i32 on Windows)
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "GetFileInformationByHandle");
        this.check_no_isolation("`GetFileInformationByHandle`")?;

        let file = this.read_handle(file)?;
        let file_information = this.deref_pointer_as(file_information, this.windows_ty_layout("BY_HANDLE_FILE_INFORMATION"))?;

        let fd = if let Handle::File(fd) = file {
            fd
        } else {
            this.invalid_handle("GetFileInformationByHandle")?
        };

        let Some(desc) = this.machine.fds.get(fd as i32) else {
            this.invalid_handle("GetFileInformationByHandle")?
        };

        let metadata = match desc.metadata()? {
            Ok(meta) => meta,
            Err(e) => {
                this.set_last_error(e)?;
                return interp_ok(this.eval_windows("c", "FALSE"))
            }
        };

        let size = metadata.len();

        let file_type = metadata.file_type();
        let attributes = if file_type.is_dir() {
            this.eval_windows_u32("c", "FILE_ATTRIBUTE_DIRECTORY")
        } else if file_type.is_file() {
            this.eval_windows_u32("c", "FILE_ATTRIBUTE_NORMAL")
        } else {
            this.eval_windows_u32("c", "FILE_ATTRIBUTE_DEVICE")
        };

        let created = extract_windows_epoch(metadata.created())?
            .unwrap_or((0, 0));
        let accessed = extract_windows_epoch(metadata.accessed())?
            .unwrap_or((0, 0));
        let written = extract_windows_epoch(metadata.modified())?
            .unwrap_or((0, 0));

        this.write_int_fields_named(
            &[("dwFileAttributes", attributes as i128)],
            &file_information,
        )?;
        write_filetime_field(this, &file_information, "ftCreationTime", created)?;
        write_filetime_field(this, &file_information, "ftLastAccessTime", accessed)?;
        write_filetime_field(this, &file_information, "ftLastWriteTime", written)?;
        this.write_int_fields_named(
            &[
                ("dwVolumeSerialNumber", 0),
                ("nFileSizeHigh", (size >> 32) as i128),
                ("nFileSizeLow", size as u32 as i128),
                ("nNumberOfLinks", 1),
                ("nFileIndexHigh", 0),
                ("nFileIndexLow", 0),
            ],
            &file_information,
        )?;

        interp_ok(this.eval_windows("c", "TRUE"))
    }

    fn DeleteFileW(
        &mut self,
        file_name: &OpTy<'tcx>, // LPCWSTR
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns BOOL (i32 on Windows)
        let this = self.eval_context_mut();
        this.assert_target_os("windows", "DeleteFileW");
        this.check_no_isolation("`DeleteFileW`")?;
        let file_name = String::from_utf16_lossy(&this.read_wide_str(this.read_pointer(file_name)?)?);
        let file_name = Path::new(&file_name);
        match std::fs::remove_file(file_name) {
            Ok(_) => interp_ok(this.eval_windows("c", "TRUE")),
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(this.eval_windows("c", "FALSE"))
            }
        }
    }

    fn NtWriteFile(
        &mut self,
        handle: &OpTy<'tcx>,          // HANDLE
        event: &OpTy<'tcx>,           // HANDLE
        apc_routine: &OpTy<'tcx>,     // PIO_APC_ROUTINE
        apc_ctx: &OpTy<'tcx>,         // PVOID
        io_status_block: &OpTy<'tcx>, // PIO_STATUS_BLOCK
        buf: &OpTy<'tcx>,             // PVOID
        n: &OpTy<'tcx>,               // ULONG
        byte_offset: &OpTy<'tcx>,     // PLARGE_INTEGER
        key: &OpTy<'tcx>,             // PULONG
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns NTSTATUS (u32 on Windows)
        let this = self.eval_context_mut();
        let handle = this.read_handle(handle)?;
        let event = this.read_handle(event)?;
        let apc_routine = this.read_pointer(apc_routine)?;
        let apc_ctx = this.read_pointer(apc_ctx)?;
        let buf = this.read_pointer(buf)?;
        let n = this.read_scalar(n)?.to_u32()?;
        let byte_offset = this.read_target_usize(byte_offset)?; // is actually a pointer
        let key = this.read_pointer(key)?;
        let io_status_block = this
            .deref_pointer_as(io_status_block, this.windows_ty_layout("IO_STATUS_BLOCK"))?;

        if event != Handle::Null {
            throw_unsup_format!(
                "`NtWriteFile` `Event` parameter is non-null, which is unsupported"
            );
        }

        if !this.ptr_is_null(apc_routine)? {
            throw_unsup_format!(
                "`NtWriteFile` `ApcRoutine` parameter is not null, which is unsupported"
            );
        }

        if !this.ptr_is_null(apc_ctx)? {
            throw_unsup_format!(
                "`NtWriteFile` `ApcContext` parameter is not null, which is unsupported"
            );
        }

        if byte_offset != 0 {
            throw_unsup_format!(
                "`NtWriteFile` `ByteOffset` parameter is non-null, which is unsupported"
            );
        }

        if !this.ptr_is_null(key)? {
            throw_unsup_format!(
                "`NtWriteFile` `Key` parameter is not null, which is unsupported"
            );
        }

        let written = match handle {
            Handle::Pseudo(pseudo @ (PseudoHandle::Stdout | PseudoHandle::Stderr)) => {
                // stdout/stderr
                let buf_cont =
                    this.read_bytes_ptr_strip_provenance(buf, Size::from_bytes(u64::from(n)))?;
                let res = if this.machine.mute_stdout_stderr {
                    Ok(buf_cont.len())
                } else if pseudo == PseudoHandle::Stdout {
                    io::Write::write(&mut io::stdout(), buf_cont)
                } else {
                    io::Write::write(&mut io::stderr(), buf_cont)
                };
                // We write at most `n` bytes, which is a `u32`, so we cannot have written more than that.
                res.ok().map(|n| u32::try_from(n).unwrap())
            }
            Handle::File(fd) => {
                let Some(desc) = this.machine.fds.get(fd as i32) else {
                    this.invalid_handle("NtWriteFile")?
                };

                let errno_layout = this.machine.layouts.u32;
                let out_place = this.allocate(errno_layout, MiriMemoryKind::Machine.into())?;
                desc.write(&desc, this.machine.communicate(), buf, n as usize, &out_place, this)?;
                let written = this.read_scalar(&out_place)?.to_u32()?;
                this.deallocate_ptr(out_place.ptr(), None, MiriMemoryKind::Machine.into())?;
                Some(written)
            }
            _ => this.invalid_handle("NtWriteFile")?,
        };

        // We have to put the result into io_status_block.
        if let Some(n) = written {
            let io_status_information =
                this.project_field_named(&io_status_block, "Information")?;
            this.write_scalar(
                Scalar::from_target_usize(n.into(), this),
                &io_status_information,
            )?;
        }

        // Return whether this was a success. >= 0 is success.
        // For the error code we arbitrarily pick 0xC0000185, STATUS_IO_DEVICE_ERROR.
        interp_ok(Scalar::from_u32(if written.is_some() { 0 } else { 0xC0000185u32 }))
    }

    fn NtReadFile(
        &mut self,
        handle: &OpTy<'tcx>,          // HANDLE
        event: &OpTy<'tcx>,           // HANDLE
        apc_routine: &OpTy<'tcx>,     // PIO_APC_ROUTINE
        apc_ctx: &OpTy<'tcx>,         // PVOID
        io_status_block: &OpTy<'tcx>, // PIO_STATUS_BLOCK
        buf: &OpTy<'tcx>,             // PVOID
        n: &OpTy<'tcx>,               // ULONG
        byte_offset: &OpTy<'tcx>,     // PLARGE_INTEGER
        key: &OpTy<'tcx>,             // PULONG
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns NTSTATUS (u32 on Windows)
        let this = self.eval_context_mut();
        let handle = this.read_handle(handle)?;
        let event = this.read_handle(event)?;
        let apc_routine = this.read_pointer(apc_routine)?;
        let apc_ctx = this.read_pointer(apc_ctx)?;
        let buf = this.read_pointer(buf)?;
        let n = this.read_scalar(n)?.to_u32()?;
        let byte_offset = this.read_target_usize(byte_offset)?; // is actually a pointer
        let key = this.read_pointer(key)?;
        let io_status_block = this
            .deref_pointer_as(io_status_block, this.windows_ty_layout("IO_STATUS_BLOCK"))?;

        if event != Handle::Null {
            throw_unsup_format!(
                "`NtWriteFile` `Event` parameter is non-null, which is unsupported"
            );
        }

        if !this.ptr_is_null(apc_routine)? {
            throw_unsup_format!(
                "`NtWriteFile` `ApcRoutine` parameter is not null, which is unsupported"
            );
        }

        if !this.ptr_is_null(apc_ctx)? {
            throw_unsup_format!(
                "`NtWriteFile` `ApcContext` parameter is not null, which is unsupported"
            );
        }

        if byte_offset != 0 {
            throw_unsup_format!(
                "`NtWriteFile` `ByteOffset` parameter is non-null, which is unsupported"
            );
        }

        if !this.ptr_is_null(key)? {
            throw_unsup_format!(
                "`NtWriteFile` `Key` parameter is not null, which is unsupported"
            );
        }

        let read = match handle {
            Handle::Pseudo(PseudoHandle::Stdin) => {
                // stdout/stderr
                let mut buf_cont = vec![0u8; n as usize];
                let res =
                    io::Read::read(&mut io::stdin(), &mut buf_cont);
                this.write_bytes_ptr(buf, buf_cont)?;
                // We write at most `n` bytes, which is a `u32`, so we cannot have written more than that.
                res.ok().map(|n| u32::try_from(n).unwrap())
            }
            Handle::File(fd) => {
                let Some(desc) = this.machine.fds.get(fd as i32) else {
                    this.invalid_handle("NtReadFile")?
                };

                let errno_layout = this.machine.layouts.u32;
                let out_place = this.allocate(errno_layout, MiriMemoryKind::Machine.into())?;
                desc.read(&desc, this.machine.communicate(), buf, n as usize, &out_place, this)?;
                let read = this.read_scalar(&out_place)?.to_u32()?;
                this.deallocate_ptr(out_place.ptr(), None, MiriMemoryKind::Machine.into())?;
                Some(read)
            }
            _ => this.invalid_handle("NtReadFile")?,
        };

        // We have to put the result into io_status_block.
        if let Some(n) = read {
            let io_status_information =
                this.project_field_named(&io_status_block, "Information")?;
            this.write_scalar(
                Scalar::from_target_usize(n.into(), this),
                &io_status_information,
            )?;
        }

        // Return whether this was a success. >= 0 is success.
        // For the error code we arbitrarily pick 0xC0000185, STATUS_IO_DEVICE_ERROR.
        interp_ok(Scalar::from_u32(if read.is_some() { 0 } else { 0xC0000185u32 }))
    }

    fn SetFilePointerEx(
        &mut self,
        file: &OpTy<'tcx>,         // HANDLE
        dist_to_move: &OpTy<'tcx>, // LARGE_INTEGER
        new_fp: &OpTy<'tcx>,       // PLARGE_INTEGER
        move_method: &OpTy<'tcx>,  // DWORD
    ) -> InterpResult<'tcx, Scalar> {
        // ^ Returns BOOL (i32 on Windows)
        let this = self.eval_context_mut();
        let file = this.read_handle(file)?;
        let dist_to_move = this.read_scalar(dist_to_move)?.to_i64()?;
        let move_method = this.read_scalar(move_method)?.to_u32()?;

        let fd = match file {
            Handle::File(fd) => fd,
            _ => this.invalid_handle("SetFilePointerEx")?,
        };

        let Some(desc) = this.machine.fds.get(fd as i32) else {
            throw_unsup_format!("`SetFilePointerEx` is only supported on file backed handles");
        };

        let file_begin = this.eval_windows_u32("c", "FILE_BEGIN");
        let file_current = this.eval_windows_u32("c", "FILE_CURRENT");
        let file_end = this.eval_windows_u32("c", "FILE_END");

        let seek = if move_method == file_begin {
            SeekFrom::Start(dist_to_move.try_into().unwrap())
        } else if move_method == file_current {
            SeekFrom::Current(dist_to_move)
        } else if move_method == file_end {
            SeekFrom::End(dist_to_move)
        } else {
            throw_unsup_format!("Invalid move method: {move_method}")
        };

        match desc.seek(this.machine.communicate(), seek)? {
            Ok(n) => {
                this.write_scalar(
                    Scalar::from_i64(n as i64),
                    &this.deref_pointer(new_fp)?,
                )?;
                interp_ok(this.eval_windows("c", "TRUE"))
            },
            Err(e) => {
                this.set_last_error(e)?;
                interp_ok(this.eval_windows("c", "FALSE"))
            }
        }
    }
}

/// Windows FILETIME is measured in 100-nanosecs since 1601
fn extract_windows_epoch<'tcx>(
    time: io::Result<SystemTime>,
) -> InterpResult<'tcx, Option<(u32, u32)>> {
    // seconds in a year * 10 million (nanoseconds/second / 100)
    const TIME_TO_EPOCH: u64 = 31_536_000 * 10_000_000;
    match time.ok() {
        Some(time) => {
            let duration = system_time_to_duration(&time)?;
            let secs = duration.as_secs() * 10_000_000;
            let nanos_hundred = (duration.subsec_nanos() / 100) as u64;
            let total = secs + nanos_hundred + TIME_TO_EPOCH;
            interp_ok(Some((total as u32, (total >> 32) as u32)))
        }
        None => interp_ok(None),
    }
}

fn write_filetime_field<'tcx>(cx: &mut MiriInterpCx<'tcx>, val: &MPlaceTy<'tcx>, name: &str, (low, high): (u32, u32)) -> InterpResult<'tcx> {
    cx.write_int_fields_named(
        &[("dwLowDateTime", low as i128), ("dwHighDateTime", high as i128)],
        &cx.project_field_named(val, name)?,
    )
}
