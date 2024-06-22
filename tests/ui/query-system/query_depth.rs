//@ build-fail
//@ error-pattern: queries overflow the depth limit!

#![recursion_limit = "64"]
type Byte = Option<Option<Option<Option< Option<Option<Option<Option<
    Option<Option<Option<Option< Option<Option<Option<Option<
        Option<Option<Option<Option< Option<Option<Option<Option<
            Option<Option<Option<Option< Option<Option<Option<Option<
                Option<Option<Option<Option< Option<Option<Option<Option<
                    Option<Option<Option<Option< Option<Option<Option<Option<
                        Option<Option<Option<Option< Option<Option<Option<Option<
                            Option<Option<Option<Option< Option<Option<Option<Option<
                                Option<Option<Option<Option< Option<Option<Option<Option<
                                    Option<Option<Option<Option< Option<Option<Option<Option<
                                        Option<Option<Option<Option< Option<Option<Option<Option<
                                            Box<String>
                                        >>>> >>>>
                                    >>>> >>>>
                                >>>> >>>>
                            >>>> >>>>
                        >>>> >>>>
                    >>>> >>>>
                >>>> >>>>
            >>>> >>>>
        >>>> >>>>
    >>>> >>>>
>>>> >>>>;

fn main() {
    println!("{}", std::mem::size_of::<Byte>());
}
