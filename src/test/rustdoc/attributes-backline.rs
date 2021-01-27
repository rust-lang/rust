#![crate_name = "foo"]

// @has 'foo/struct.Fly.html'
// @has - '//h4[@id="method.drosophilae"]//span[@class="docblock attributes"]' ''
// @has - '//h4[@id="method.drosophilae"]//span[@class="docblock attributes"]' \
//    'Just a sentence with a backline in the middle'
// @!has - '//h4[@id="method.drosophilae"]//span[@class="doblock attributes"]' '\\'
pub struct Fly;

impl Fly {
    #[must_use = "Just a sentence with a backline in \
                  the middle"]
    pub fn drosophilae(&self) {}
}
