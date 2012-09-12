// error-pattern:runned an unexported test
// compile-flags:--test

extern mod std;

mod m {
    export exported;

    fn exported() { }

    #[test]
    fn unexported() { fail ~"runned an unexported test"; }
}
