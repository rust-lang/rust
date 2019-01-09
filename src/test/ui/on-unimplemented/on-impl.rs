// Test if the on_unimplemented message override works

#![feature(on_unimplemented)]


#[rustc_on_unimplemented = "invalid"]
trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, index: Idx) -> &Self::Output;
}

#[rustc_on_unimplemented = "a usize is required to index into a slice"]
impl Index<usize> for [i32] {
    type Output = i32;
    fn index(&self, index: usize) -> &i32 {
        &self[index]
    }
}


fn main() {
    Index::<u32>::index(&[1, 2, 3] as &[i32], 2u32);
    //~^ ERROR E0277
    //~| ERROR E0277
}
