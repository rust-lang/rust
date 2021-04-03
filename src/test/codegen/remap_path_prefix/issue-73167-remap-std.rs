// ignore-windows

// compile-flags: -g  -C no-prepopulate-passes --remap-path-prefix=/=/the/root/

// Here we check that imported code from std has their path remapped

// CHECK: !DIFile(filename: "{{/the/root/.*/library/std/src/panic.rs}}"
fn main() {
    std::thread::spawn(|| {
        println!("hello");
    });
}
