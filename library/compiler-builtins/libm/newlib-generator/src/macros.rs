macro_rules! f32 {
    ($($fun:ident,)+) => {
        $(
            let fun = stringify!($fun);

            fs::create_dir_all("math/src")?;

            let main = format!("
#![no_main]
#![no_std]

#[macro_use]
extern crate qemu_arm_rt as rt;

use core::u32;

use rt::{{io, process}};

entry!(main);

fn main() {{
    run().unwrap_or_else(|e| {{
        eprintln!(\"error: {{}}\", e);
        process::exit(1);
    }})
}}

fn run() -> Result<(), usize> {{
    #[link(name = \"m\")]
    extern \"C\" {{
        fn {0}(_: f32) -> f32;
    }}

    let mut buf = [0; 4];
    while let Ok(()) = io::Stdin.read_exact(&mut buf) {{
        let x = f32::from_bits(u32::from_bytes(buf));
        let y = unsafe {{ {0}(x) }};

        io::Stdout.write_all(&y.to_bits().to_bytes())?;
    }}

    Ok(())
}}

#[no_mangle]
pub fn __errno() -> *mut i32 {{
    static mut ERRNO: i32 = 0;
    unsafe {{ &mut ERRNO }}
}}
", fun);

            File::create("math/src/main.rs")?.write_all(main.as_bytes())?;

            assert!(
                Command::new("cross")
                    .args(&["build", "--target", "thumbv7em-none-eabi", "--release"])
                    .current_dir("math")
                    .status()?
                .success()
            );

            let mut qemu = Command::new("qemu-arm")
                .arg("math/target/thumbv7em-none-eabi/release/math")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()?;

            qemu.stdin.as_mut().take().unwrap().write_all(F32)?;

            let output = qemu.wait_with_output()?;

            File::create(concat!("bin/output/newlib.", stringify!($fun)))?
                .write_all(&output.stdout)?;
        )+
    }
}

macro_rules! f32f32 {
    ($($fun:ident,)+) => {
        $(
            let fun = stringify!($fun);

            fs::create_dir_all("math/src")?;

            let main = format!("
#![no_main]
#![no_std]

#[macro_use]
extern crate qemu_arm_rt as rt;

use core::u32;

use rt::{{io, process}};

entry!(main);

fn main() {{
    run().unwrap_or_else(|e| {{
        eprintln!(\"error: {{}}\", e);
        process::exit(1);
    }})
}}

fn run() -> Result<(), usize> {{
    #[link(name = \"m\")]
    extern \"C\" {{
        fn {0}(_: f32, _: f32) -> f32;
    }}

    let mut chunk = [0; 8];
    while let Ok(()) = io::Stdin.read_exact(&mut chunk) {{
        let mut buf = [0; 4];
        buf.copy_from_slice(&chunk[..4]);
        let x0 = f32::from_bits(u32::from_bytes(buf));

        buf.copy_from_slice(&chunk[4..]);
        let x1 = f32::from_bits(u32::from_bytes(buf));

        let y = unsafe {{ {0}(x0, x1) }};

        io::Stdout.write_all(&y.to_bits().to_bytes())?;
    }}

    Ok(())
}}

#[no_mangle]
pub fn __errno() -> *mut i32 {{
    static mut ERRNO: i32 = 0;
    unsafe {{ &mut ERRNO }}
}}
", fun);

            File::create("math/src/main.rs")?.write_all(main.as_bytes())?;

            assert!(
                Command::new("cross")
                    .args(&["build", "--target", "thumbv7em-none-eabi", "--release"])
                    .current_dir("math")
                    .status()?
                .success()
            );

            let mut qemu = Command::new("qemu-arm")
                .arg("math/target/thumbv7em-none-eabi/release/math")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()?;

            qemu.stdin.as_mut().take().unwrap().write_all(F32)?;

            let output = qemu.wait_with_output()?;

            File::create(concat!("bin/output/newlib.", stringify!($fun)))?
                .write_all(&output.stdout)?;
        )+
    }
}

macro_rules! f32f32f32 {
    ($($fun:ident,)+) => {
        $(
            let fun = stringify!($fun);

            fs::create_dir_all("math/src")?;

            let main = format!("
#![no_main]
#![no_std]

#[macro_use]
extern crate qemu_arm_rt as rt;

use core::u32;

use rt::{{io, process}};

entry!(main);

fn main() {{
    run().unwrap_or_else(|e| {{
        eprintln!(\"error: {{}}\", e);
        process::exit(1);
    }})
}}

fn run() -> Result<(), usize> {{
    #[link(name = \"m\")]
    extern \"C\" {{
        fn {0}(_: f32, _: f32, _: f32) -> f32;
    }}

    let mut chunk = [0; 12];
    while let Ok(()) = io::Stdin.read_exact(&mut chunk) {{
        let mut buf = [0; 4];
        buf.copy_from_slice(&chunk[..4]);
        let x0 = f32::from_bits(u32::from_bytes(buf));

        buf.copy_from_slice(&chunk[4..8]);
        let x1 = f32::from_bits(u32::from_bytes(buf));

        buf.copy_from_slice(&chunk[8..]);
        let x2 = f32::from_bits(u32::from_bytes(buf));

        let y = unsafe {{ {0}(x0, x1, x2) }};

        io::Stdout.write_all(&y.to_bits().to_bytes())?;
    }}

    Ok(())
}}

#[no_mangle]
pub fn __errno() -> *mut i32 {{
    static mut ERRNO: i32 = 0;
    unsafe {{ &mut ERRNO }}
}}
", fun);

            File::create("math/src/main.rs")?.write_all(main.as_bytes())?;

            assert!(
                Command::new("cross")
                    .args(&["build", "--target", "thumbv7em-none-eabi", "--release"])
                    .current_dir("math")
                    .status()?
                .success()
            );

            let mut qemu = Command::new("qemu-arm")
                .arg("math/target/thumbv7em-none-eabi/release/math")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()?;

            qemu.stdin.as_mut().take().unwrap().write_all(F32)?;

            let output = qemu.wait_with_output()?;

            File::create(concat!("bin/output/newlib.", stringify!($fun)))?
                .write_all(&output.stdout)?;
        )+
    }
}
