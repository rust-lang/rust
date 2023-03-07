fn main() {
    enum Test {
        Very //~ HELP missing `,`
        Bad(usize) //~ HELP missing `,`
        //~^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found `Bad`
        Stuff { a: usize } //~ HELP missing `,`
        //~^ ERROR expected one of `,`, `=`, or `}`, found `Stuff`
        Here
        //~^ ERROR expected one of `,`, `=`, or `}`, found `Here`
    }
}
