#![feature(thread_local)]

#[thread_local]
static FOO: u8 = 3;

fn main() {
    let a = &FOO;
    //~^ ERROR borrowed value does not live long enough
    //~| does not live long enough
    //~| NOTE borrowed value must be valid for the static lifetime

    std::thread::spawn(move || {
        println!("{}", a);
    });
}
//~^ NOTE borrowed value only lives until here
