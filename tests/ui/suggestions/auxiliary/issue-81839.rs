//@ edition:2018

pub struct Test {}

impl Test {
    pub async fn answer_str(&self, _s: &str) -> Test {
        Test {}
    }
}
