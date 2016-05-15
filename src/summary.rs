#[must_use]
#[derive(Debug, Clone)]
pub struct Summary {
    // Encountered e.g. an IO error.
    has_operational_errors: bool,

    // Failed to reformat code because of parsing errors.
    has_parsing_errors: bool,

    // Code is valid, but it is impossible to format it properly.
    has_formatting_errors: bool,
}

impl Summary {
    pub fn new() -> Summary {
        Summary {
            has_operational_errors: false,
            has_parsing_errors: false,
            has_formatting_errors: false,
        }
    }

    pub fn has_operational_errors(&self) -> bool {
        self.has_operational_errors
    }

    pub fn has_parsing_errors(&self) -> bool {
        self.has_parsing_errors
    }

    pub fn has_formatting_errors(&self) -> bool {
        self.has_formatting_errors
    }

    pub fn add_operational_error(&mut self) {
        self.has_operational_errors = true;
    }

    pub fn add_parsing_error(&mut self) {
        self.has_parsing_errors = true;
    }

    pub fn add_formatting_error(&mut self) {
        self.has_formatting_errors = true;
    }

    pub fn has_no_errors(&self) -> bool {
        !(self.has_operational_errors || self.has_parsing_errors || self.has_formatting_errors)
    }

    pub fn add(&mut self, other: Summary) {
        self.has_operational_errors |= other.has_operational_errors;
        self.has_formatting_errors |= other.has_formatting_errors;
        self.has_parsing_errors |= other.has_parsing_errors;
    }
}
