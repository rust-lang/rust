pub trait X {
    fn a(&self) -> &'static str;
    fn bcd(&self,
           c: &str, // comment on this arg
           d: u16, // comment on this arg
           e: &Vec<String> // comment on this arg
           ) -> Box<Q>;
}
