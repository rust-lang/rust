// ICE #105210 self.lines.iter().all(|r| !r.iter().any(|sc| sc.chr == \'\\t\'))
// ignore-tidy-tab
//@ edition:2021
pub fn main() {}

fn box () {
 (( h (const {( default ( await ( await (	(move {await((((}}
 //~^ ERROR mismatched closing delimiter: `}`
 //~^^ ERROR mismatched closing delimiter: `}`
//~ ERROR this file contains an unclosed delimiter
