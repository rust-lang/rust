trait A: Sized {
    type N;
    fn x() ->
        Self<
          N= //~ ERROR associated type bindings are not allowed here
          Self::N> {
        loop {}
    }
    fn y(&self) ->
        std
           <N=()> //~ ERROR associated type bindings are not allowed here
           ::option::Option<()>
    { None }
    fn z(&self) ->
        u32<N=()> //~ ERROR associated type bindings are not allowed here
    { 42 }

}

fn main() {
}
