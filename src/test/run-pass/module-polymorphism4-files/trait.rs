trait says {
    fn says() -> ~str;
}

impl T: says {

    // 'animal' and 'talk' functions are implemented by the module
    // instantiating the talky trait. They are 'abstract'
    fn says() -> ~str {
        animal() + ~" says '" + talk(self) + ~"'"
    }

}
