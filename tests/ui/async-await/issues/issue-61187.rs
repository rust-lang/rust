//@ check-pass
//@ edition:2018

fn main() {}

async fn response(data: Vec<u8>) {
    data.reverse(); //~ WARNING E0596
}
