//@ run-rustfix
pub struct LipogramCorpora {
    selections: Vec<(char, Option<String>)>,
}

impl LipogramCorpora {
    pub fn validate_all(&mut self) -> Result<(), char> {
        for selection in &self.selections {
            if selection.1.is_some() {
                if selection.1.unwrap().contains(selection.0) {
                //~^ ERROR cannot move out of `selection.1`
                    return Err(selection.0);
                }
            }
        }
        Ok(())
    }
}

pub struct LipogramCorpora2 {
    selections: Vec<(char, Result<String, String>)>,
}

impl LipogramCorpora2 {
    pub fn validate_all(&mut self) -> Result<(), char> {
        for selection in &self.selections {
            if selection.1.is_ok() {
                if selection.1.unwrap().contains(selection.0) {
                //~^ ERROR cannot move out of `selection.1`
                    return Err(selection.0);
                }
            }
        }
        Ok(())
    }
}

fn main() {}
