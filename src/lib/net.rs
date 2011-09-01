import vec;
import uint;

tag ip_addr { ipv4(u8, u8, u8, u8); }

fn format_addr(ip: ip_addr) -> istr {
    alt ip {
      ipv4(a, b, c, d) {
        #ifmt["%u.%u.%u.%u", a as uint, b as uint, c as uint, d as uint]
      }
      _ { fail "Unsupported address type"; }
    }
}

fn parse_addr(ip: &istr) -> ip_addr {
    let parts = vec::map(
        { |&s| uint::from_str(s) },
        istr::split(ip, ~"."[0]));
    if vec::len(parts) != 4u { fail "Too many dots in IP address"; }
    for i in parts { if i > 255u { fail "Invalid IP Address part."; } }
    ipv4(parts[0] as u8, parts[1] as u8, parts[2] as u8, parts[3] as u8)
}
