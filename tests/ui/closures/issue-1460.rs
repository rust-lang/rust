//@ run-pass


pub fn main() {
    {|i: u32| if 1 == i { }}; //~ WARN unused closure that must be used
}
