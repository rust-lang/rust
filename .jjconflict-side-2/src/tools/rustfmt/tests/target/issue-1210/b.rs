// rustfmt-format_strings: true
// rustfmt-max_width: 50

impl Foo {
    fn cxx(&self, target: &str) -> &Path {
        match self.cxx.get(target) {
            Some(p) => p.path(),
            None => panic!(
                "\ntarget `{}`: is not, \
                 configured as a host,
                            only as a target\n\n",
                target
            ),
        }
    }
}
