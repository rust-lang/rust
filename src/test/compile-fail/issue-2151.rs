fn main() {
    for vec::each(fail) |i| {
        log (debug, i * 2);
        //~^ ERROR the type of this value must be known
   };
}