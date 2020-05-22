pub enum ErrorHandled {
    Reported,
    TooGeneric,
}

impl ErrorHandled {
    pub fn assert_reported(self) {
        match self {
            ErrorHandled::Reported => {}}
                                     //^~ ERROR block is empty, you might have not meant to close it
            ErrorHandled::TooGeneric => panic!(),
        }
    }
} //~ ERROR unexpected closing delimiter: `}`
