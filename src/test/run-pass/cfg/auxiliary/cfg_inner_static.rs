// this used to just ICE on compiling
pub fn foo() {
    if cfg!(foo) {
        static a: isize = 3;
        a
    } else { 3 };
}
