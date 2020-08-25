fn main() {
    enum Test {
        Very //~ HELP try adding a comma
        Bad(usize) //~ HELP try adding a comma
        //~^ ERROR expected one of `(`, `,`, `=`, or `{`, found `Bad`
        Stuff { a: usize } //~ HELP try adding a comma
        //~^ ERROR expected one of `,` or `=`, found `Stuff`
        Here
        //~^ ERROR expected one of `,` or `=`, found `Here`
    }
}
