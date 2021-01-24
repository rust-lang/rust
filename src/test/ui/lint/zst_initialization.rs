// check-pass

enum Never {}

fn main() {
    unsafe {
        std::mem::transmute::<(), Never>(());
        //~^ WARNING: the type `Never` does not permit zero-initialization
        std::mem::transmute::<Option<Never>, Never>(None);
        //~^ WARNING: the type `Never` does not permit initialization
    }
}
