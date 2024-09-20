// rustfmt-style_edition: 2024
// rustfmt-wrap_comments: true

pub const IFF_MULTICAST: ::c_int = 0x0000000800; // Supports multicast
// Multicast using broadcst. add.

pub const SQ_CRETAB: u16 = 0x000e; // CREATE TABLE
pub const SQ_DRPTAB: u16 = 0x000f; // DROP TABLE
pub const SQ_CREIDX: u16 = 0x0010; // CREATE INDEX
//const SQ_DRPIDX: u16 = 0x0011;	// DROP INDEX
//const SQ_GRANT: u16 = 0x0012;	// GRANT
//const SQ_REVOKE: u16 = 0x0013;	// REVOKE

fn foo() {
    let f = bar(); // Donec consequat mi. Quisque vitae dolor. Integer lobortis. Maecenas id nulla. Lorem.
    // Id turpis. Nam posuere lectus vitae nibh. Etiam tortor orci, sagittis malesuada, rhoncus quis, hendrerit eget, libero. Quisque commodo nulla at nunc. Mauris consequat, enim vitae venenatis sollicitudin, dolor orci bibendum enim, a sagittis nulla nunc quis elit. Phasellus augue. Nunc suscipit, magna tincidunt lacinia faucibus, lacus tellus ornare purus, a pulvinar lacus orci eget nibh.  Maecenas sed nibh non lacus tempor faucibus. In hac habitasse platea dictumst. Vivamus a orci at nulla tristique condimentum. Donec arcu quam, dictum accumsan, convallis accumsan, cursus sit amet, ipsum.  In pharetra sagittis nunc.
    let b = baz();

    let normalized = self.ctfont.all_traits().normalized_weight(); // [-1.0, 1.0]
    // TODO(emilio): It may make sense to make this range [.01, 10.0], to align with css-fonts-4's range of [1, 1000].
}
