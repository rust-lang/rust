# armv7-sony-vita-newlibeabihf

**Tier: 3**

This tier supports the ARM Cortex A9 processor running on a PlayStation Vita console. `armv7-vita-newlibeabihf` aims to have support for `std` crate using `newlib` as a bridge.

Rust support for this target is not affiliated with Sony, and is not derived
from nor used with any official Sony SDK.

## Designated Developers

* [@amg98](https://github.com/amg98)
* [@nikarh](https://github.com/nikarh)

## Requirements

This target is cross-compiled, and requires installing [VITASDK](https://vitasdk.org/) toolchain on your system. Dynamic linking is not supported.

`#![no_std]` crates can be built using `build-std` to build `core`, and optionally
`alloc`, and `panic_abort`.

`std` is partially supported, but mostly works. Some APIs are unimplemented
and will simply return an error, such as `std::process`. An allocator is provided
by default.

In order to support some APIs, binaries must be linked against `libc` written
for the target, using a linker for the target. These are provided by the
VITASDK toolchain.

This target generates binaries in the ELF format.

## Building

Rust does not ship pre-compiled artifacts for this target. You can use `build-std` flag to build binaries with `std`:

```sh
cargo build -Z build-std=std,panic_abort --target=armv7-sony-vita-newlibeabihf --release
```

## Cross-compilation

This target can be cross-compiled from `x86_64` on either Windows, MacOS or Linux systems. Other hosts are not supported for cross-compilation.

## Testing

Currently there is no support to run the rustc test suite for this target.

## Building and Running Rust Programs

`std` support for this target relies on newlib. In order to work, newlib must be initialized correctly. The easiest way to achieve this with VITASDK newlib implementation is by compiling your program as a staticlib with and exposing your main function from rust to `_init` function in `crt0`.

Add this to your `Cargo.toml`:

```toml
[lib]
crate-type = ["staticlib"]

[profile.release]
panic = 'abort'
lto = true
opt-level = 3
```

Your entrypoint should look roughly like this, `src/lib.rs`:
```rust,ignore,no_run
#[used]
#[export_name = "_newlib_heap_size_user"]
pub static _NEWLIB_HEAP_SIZE_USER: u32 = 100 * 1024 * 1024; // Default heap size is only 32mb, increase it to something suitable for your application

#[no_mangle]
pub extern "C" fn main() {
    println!("Hello, world!");
}
```

To test your developed rust programs on PlayStation Vita, first you must correctly link and package your rust staticlib. These steps can be preformed using tools available in VITASDK, and can be automated using tools like `cargo-make`.

First, set up environment variables for `VITASDK`, and it's binaries:

```sh
export VITASDK=/opt/vitasdk
export PATH=$PATH:$VITASDK/bin
```

Use the example below as a template for your project:

```toml
[env]
TITLE = "Rust Hello World"
TITLEID = "RUST00001"
# Add other libs required by your project here
LINKER_LIBS = "-lpthread -lm -lmathneon"

# At least a "sce_sys" folder should be place there for app metadata (title, icons, description...)
# You can find sample assets for that on $VITASDK/share/gcc-arm-vita-eabi/samples/hello_world/sce_sys/
STATIC_DIR = "static"   # Folder where static assets should be placed (sce_sys folder is at $STATIC_DIR/sce_sys)
CARGO_TARGET_DIR = { script = ["echo ${CARGO_TARGET_DIR:=target}"] }
RUST_TARGET = "armv7-sony-vita-newlibeabihf"
CARGO_OUT_DIR = "${CARGO_TARGET_DIR}/${RUST_TARGET}/release"

TARGET_LINKER = "arm-vita-eabi-gcc"
TARGET_LINKER_FLAGS = "-Wl,-q"

[tasks.build]
description = "Build the project using `cargo` as a static lib."
command = "cargo"
args = ["build", "-Z", "build-std=std,panic_abort", "--target=armv7-sony-vita-newlibeabihf", "--release"]

[tasks.link]
description = "Build an ELF executable using the `vitasdk` linker."
dependencies = ["build"]
script = [
    """
    ${TARGET_LINKER} ${TARGET_LINKER_FLAGS} \
        -L"${CARGO_OUT_DIR}" \
        -l"${CARGO_MAKE_CRATE_FS_NAME}" \
        ${LINKER_LIBS} \
        -o"${CARGO_OUT_DIR}/${CARGO_MAKE_CRATE_NAME}.elf"
    """
]

[tasks.strip]
description = "Strip the produced ELF executable."
dependencies = ["link"]
command = "arm-vita-eabi-strip"
args = ["-g", '${CARGO_OUT_DIR}/${CARGO_MAKE_CRATE_FS_NAME}.elf']

[tasks.velf]
description = "Build an VELF executable from the obtained ELF file."
dependencies = ["strip"]
command = "vita-elf-create"
args = ['${CARGO_OUT_DIR}/${CARGO_MAKE_CRATE_NAME}.elf', '${CARGO_OUT_DIR}/${CARGO_MAKE_CRATE_NAME}.velf']

[tasks.eboot-bin]
description = "Build an `eboot.bin` file from the obtained VELF file."
dependencies = ["velf"]
command = "vita-make-fself"
args = ["-s", '${CARGO_OUT_DIR}/${CARGO_MAKE_CRATE_NAME}.velf', '${CARGO_OUT_DIR}/eboot.bin']

[tasks.param-sfo]
description = "Build the `param.sfo` manifest using with given TITLE and TITLEID."
command = "vita-mksfoex"
args = ["-s", 'TITLE_ID=${TITLEID}', '${TITLE}', '${CARGO_OUT_DIR}/param.sfo']

[tasks.manifest]
description = "List all static resources into a manifest file."
script = [
  'mkdir -p "${CARGO_OUT_DIR}"',
  '''
  if [ -d "${STATIC_DIR}" ]; then
    find "${STATIC_DIR}" -type f > "${CARGO_OUT_DIR}/MANIFEST"
  else
    touch "${CARGO_OUT_DIR}/MANIFEST"
  fi
  '''
]

[tasks.vpk]
description = "Build a VPK distribution of the project executable and resources."
dependencies = ["eboot-bin", "param-sfo", "manifest"]
script_runner = "@rust"
script = [
    '''
    use std::io::BufRead;
    use std::fs::File;

    fn main() {

      let crate_name = env!("CARGO_MAKE_CRATE_NAME");
      let static_dir = env!("STATIC_DIR");
      let out_dir = std::path::PathBuf::from(env!("CARGO_OUT_DIR"));

      let mut cmd = ::std::process::Command::new("vita-pack-vpk");
      cmd.arg("-s").arg(out_dir.join("param.sfo"));
      cmd.arg("-b").arg(out_dir.join("eboot.bin"));

      // Add files from MANIFEST
      if let Ok(file) = File::open(out_dir.join("MANIFEST")) {
          let mut reader = ::std::io::BufReader::new(file);
          let mut lines = reader.lines();
          while let Some(Ok(line)) = lines.next() {
              let p1 = ::std::path::PathBuf::from(line);            // path on FS
              let p2 = p1.strip_prefix(static_dir).unwrap();        // path in VPK
              cmd.arg("--add").arg(format!("{}={}", p1.display(), p2.display()));
          }
      }

      cmd.arg(out_dir.join(format!("{}.vpk", crate_name)))
        .output()
        .expect("command failed.");
    }
    '''
]
```

After running the above script, you should be able to get a *.vpk file in the same folder your *.elf executable resides. Now you can pick it and install it on your own PlayStation Vita using, or you can use an [Vita3K](https://vita3k.org/) emulator.
