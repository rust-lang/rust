//@ edition:2018

fn main() {}

async fn response(data: Vec<u8>) {
    data.reverse(); //~ ERROR E0596
}
