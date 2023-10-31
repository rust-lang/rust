#!/usr/bin/env bash
#![deny(unsafe_code)] /*This line is ignored by bash
# This block is ignored by rustc
echo "Warning: y.rs is a deprecated alias for y.sh" 1>&2
exec ./y.sh "$@"
*/
