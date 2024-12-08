//@ build-pass
//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+GVN

fn main() {
    let variant: Option<u32> = None;
    let transmuted: u64 = unsafe { std::mem::transmute(variant) };
    println!("{transmuted}");
}
