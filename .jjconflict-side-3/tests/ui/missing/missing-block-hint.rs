fn main() {
    {
        if (foo) => {} //~ ERROR expected `{`, found `=>`
    }
    {
        if (foo)
            bar; //~ ERROR expected `{`, found `bar`
    }
}
