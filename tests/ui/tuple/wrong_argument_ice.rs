use std::collections::VecDeque;

pub struct BuildPlanBuilder {
    acc: VecDeque<(String, String)>,
    current_provides: String,
    current_requires: String,
}

impl BuildPlanBuilder {
    pub fn or(&mut self) -> &mut Self {
        self.acc.push_back(self.current_provides, self.current_requires);
        //~^ ERROR method takes 1 argument but 2 arguments were supplied
        self
    }
}

fn main() {}
