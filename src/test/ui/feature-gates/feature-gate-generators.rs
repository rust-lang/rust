fn main() {
    yield true; //~ ERROR yield syntax is experimental
                //~^ ERROR yield statement outside of generator literal
}
