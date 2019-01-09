use zoo::fly; //~ ERROR: function `fly` is private

mod zoo {
    fn fly() {}
}


fn main() {
    fly();
}
