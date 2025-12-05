struct Checked<const F: fn(usize) -> bool>;
//~^ ERROR function pointers as const generic parameters is forbidden
fn not_one(val: usize) -> bool { val != 1 }
const _: Checked<not_one> = Checked::<not_one>;
fn main() {}
