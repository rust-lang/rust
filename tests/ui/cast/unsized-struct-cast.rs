pub struct Data([u8]);

fn main(){
    const _: *const Data = &[] as *const Data;
    //~^ ERROR: casting `&[_; 0]` as `*const Data` is invalid
}
