// run-pass
#[repr(C)]
pub enum CPOption<T> {
    PSome(T),
}

fn main() {
  println!("sizeof CPOption<i32> {}", std::mem::size_of::<CPOption<i32>>());
}
