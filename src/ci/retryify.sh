#!/usr/bin/env bash
# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -e

EXECUTABLES=("clang" "cc" "curl" "git")
WRAPPER_PATH=`mktemp -d`
export PATH=$WRAPPER_PATH:$PATH

for i in "${EXECUTABLES[@]}"
do

FILE=$WRAPPER_PATH/$i
tee $FILE <<TEMPLATE > /dev/null
#!/usr/bin/env bash
WP='$WRAPPER_PATH'
PATH=\${PATH//":\$WP:"/":"} # delete any instances in the middle
PATH=\${PATH/#"\$WP:"/} # delete any instance at the beginning
PATH=\${PATH/%":\$WP"/} # delete any instance in the at the end
export PATH
for iter in 1 2 3
do
    $i \$@ && break
done
TEMPLATE

chmod +x $FILE

done

echo "export PATH=$PATH"
