trait says {
    fn says() -> ~str;
}

impl talky of says for T {

    // 'animal' and 'talk' functions are implemented by the module
    // instantiating the talky trait. They are 'abstract'
    fn says() -> ~str {
        animal() + ~" says '" + talk(self) + ~"'"
    }

}
