// check-pass

struct CNFParser {
    token: char,
}

impl CNFParser {
    fn is_whitespace(c: char) -> bool {
        c == ' ' || c == '\n'
    }

    fn consume_whitespace(&mut self) {
        self.consume_while(&(CNFParser::is_whitespace))
    }

    fn consume_while(&mut self, p: &dyn Fn(char) -> bool) {
        while p(self.token) {
            return
        }
    }
}

fn main() {}
