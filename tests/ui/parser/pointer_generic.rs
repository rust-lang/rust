fn size<*T>(_v: *const T) {
    //~^ ERROR pointer types are not allowed in generic parameter lists
    println!("{:?}", std::mem::size_of::<T>());
}

fn main() {
    size(0 as *const u8);
}
