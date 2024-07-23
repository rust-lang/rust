//@ edition:2018

fn main() {
    let s = ::xcrate::S;
    //~^ ERROR cannot find item `xcrate`
}
