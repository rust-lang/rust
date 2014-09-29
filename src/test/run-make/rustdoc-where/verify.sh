#!/bin/bash
set -e

# $1 is the TMPDIR
DOC=$1/doc/foo

grep "Alpha.*where.*A:.*MyTrait" $DOC/struct.Alpha.html > /dev/null
echo "Alpha"
grep "Bravo.*where.*B:.*MyTrait" $DOC/trait.Bravo.html > /dev/null
echo "Bravo"
grep "charlie.*where.*C:.*MyTrait" $DOC/fn.charlie.html > /dev/null
echo "Charlie"
grep "impl.*Delta.*where.*D:.*MyTrait" $DOC/struct.Delta.html > /dev/null
echo "Delta"
grep "impl.*MyTrait.*for.*Echo.*where.*E:.*MyTrait" $DOC/struct.Echo.html > /dev/null
echo "Echo"
grep "impl.*MyTrait.*for.*Foxtrot.*where.*F:.*MyTrait" $DOC/enum.Foxtrot.html > /dev/null
echo "Foxtrot"

# check "Implementors" section of MyTrait
grep "impl.*MyTrait.*for.*Echo.*where.*E:.*MyTrait" $DOC/trait.MyTrait.html > /dev/null
grep "impl.*MyTrait.*for.*Foxtrot.*where.*F:.*MyTrait" $DOC/trait.MyTrait.html > /dev/null
echo "Implementors OK"
