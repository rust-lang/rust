use std::cell::RefCell;

fn main() {
    thread_local! {
        static S: RefCell<String> = RefCell::default();
    }

    S.with(|x| *x.borrow_mut() = "pika pika".to_string());
    S.with(|x| println!("{}", x.borrow()));
}
