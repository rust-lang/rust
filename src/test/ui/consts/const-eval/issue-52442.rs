fn main() {
    [();  { &loop { break } as *const _ as usize } ];
    //~^ ERROR pointers cannot be cast to integers during const eval
}
