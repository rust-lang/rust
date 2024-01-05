static S: i32 = 0;
static mut S_MUT: i32 = 0;

const C1: &i32 = &S; //~ERROR:  referencing statics in constants is unstable
const C1_READ: () = {
    assert!(*C1 == 0);
};
const C2: *const i32 = unsafe { std::ptr::addr_of!(S_MUT) }; //~ERROR:  referencing statics in constants is unstable
//~^ERROR:  referencing statics in constants is unstable

fn main() {
}
