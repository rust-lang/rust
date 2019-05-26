// edition:2018
#![feature(async_await)]

fn main() {
}

async fn response(data: Vec<u8>) {
    data.reverse(); //~ ERROR E0596
}
