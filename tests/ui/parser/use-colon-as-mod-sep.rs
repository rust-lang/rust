// Recover from using a colon as a path separator.

use std::process:Command;
//~^ ERROR expected `::`, found `:`
use std:fs::File;
//~^ ERROR expected `::`, found `:`
use std:collections:HashMap;
//~^ ERROR expected `::`, found `:`
//~| ERROR expected `::`, found `:`

fn main() { }
