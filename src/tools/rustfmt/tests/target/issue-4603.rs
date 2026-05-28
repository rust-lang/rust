// Formatting when original macro snippet is used

// Original issue #4603 code
#![feature(or_patterns)]
macro_rules! t_or_f {
    () => {
        (true // some comment
        | false)
    };
}

// Other test cases variations
macro_rules! RULES {
    () => {
        (
		xxxxxxx // COMMENT
        | yyyyyyy
        )
    };
}
macro_rules! RULES {
    () => {
        (xxxxxxx // COMMENT
            | yyyyyyy)
    };
}

fn main() {
    macro_rules! RULES {
		() => {
			(xxxxxxx // COMMENT
			| yyyyyyy)
		};
	}
}

macro_rules! RULES {
    () => {
        (xxxxxxx /* COMMENT */ | yyyyyyy)
    };
}
macro_rules! RULES {
    () => {
        (xxxxxxx /* COMMENT */
        | yyyyyyy)
    };
}
