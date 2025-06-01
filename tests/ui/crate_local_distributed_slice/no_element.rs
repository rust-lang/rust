#![feature(crate_local_distributed_slice)]

#[distributed_slice(crate)]
const MEOWS: [&str; _];

distributed_slice_element!(MEOWS,);
//~^ ERROR expected expression, found end of macro argument
distributed_slice_element!(MEOWS);
//~^ ERROR expected one of `,` or `::`, found `<eof>`
distributed_slice_element!();
//~^ ERROR expected identifier, found `<eof>`
distributed_slice_element!(MEOWS, "mew", ); // trailing comma ok

fn main() {
    println!("{MEOWS:?}");
}
