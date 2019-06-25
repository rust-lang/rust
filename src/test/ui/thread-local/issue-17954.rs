#![feature(thread_local)]

#[thread_local]
static FOO: u8 = 3;

fn main() {
    let a = &FOO;
    //~^ ERROR thread-local variable borrowed past end of function
    //~| NOTE thread-local variables cannot be borrowed beyond the end of the function

    std::thread::spawn(move || {
        println!("{}", a);
    });
}
//~^ NOTE end of enclosing function is here
