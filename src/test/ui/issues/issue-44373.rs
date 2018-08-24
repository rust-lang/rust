static FOO: u32 = 50;

fn main() {
    let _val: &'static [&'static u32] = &[&FOO]; //~ ERROR borrowed value does not live long enough
}
