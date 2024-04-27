//@ build-fail

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
//~^ ERROR: queries overflow the depth limit!
    println!("{}", std::mem::size_of::<Byte>());
}
