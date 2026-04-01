pub enum ErrorHandled {
    Reported,
    TooGeneric,
}

impl ErrorHandled {
    pub fn assert_reported(self) {
        match self {
            //~^ NOTE this delimiter might not be properly closed...
            ErrorHandled::Reported => {}}
                                     //~^ NOTE block is empty, you might have not meant to close it
                                     //~| NOTE as it matches this but it has different indentation
            ErrorHandled::TooGeneric => panic!(),
        }
    }
}
//~^ ERROR unexpected closing delimiter: `}`
//~| NOTE unexpected closing delimiter
