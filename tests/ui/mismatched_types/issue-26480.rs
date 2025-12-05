extern "C" {
    fn write(fildes: i32, buf: *const i8, nbyte: u64) -> i64;
}

#[inline(always)]
fn size_of<T>(_: T) -> usize {
    ::std::mem::size_of::<T>()
}

macro_rules! write {
    ($arr:expr) => {{
        #[allow(non_upper_case_globals)]
        const stdout: i32 = 1;
        unsafe {
            write(stdout, $arr.as_ptr() as *const i8,
                  $arr.len() * size_of($arr[0])); //~ ERROR mismatched types
        }
    }}
}

macro_rules! cast {
    ($x:expr) => ($x as ()) //~ ERROR non-primitive cast
}

fn main() {
    let hello = ['H', 'e', 'y'];
    write!(hello);
    cast!(2);
}
