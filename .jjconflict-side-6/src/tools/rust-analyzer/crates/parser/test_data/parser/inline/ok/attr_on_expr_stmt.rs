fn foo() {
    #[A] foo();
    #[B] bar!{}
    #[C] #[D] {}
    #[D] return ();
}
