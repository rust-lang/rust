// rustfmt-reorder_impl_items: true

// The ordering of the following impl items should be idempotent.
impl<'a> Command<'a> {
    pub fn send_to(&self, w: &mut io::Write) -> io::Result<()> {
        match self {
            &Command::Data(ref c) => c.send_to(w),
            &Command::Vrfy(ref c) => c.send_to(w),
        }
    }

    pub fn parse(arg: &[u8]) -> Result<Command, ParseError> {
        nom_to_result(command(arg))
    }
}
