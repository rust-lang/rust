#!/bin/sh

source="$1/index.rs"
index="$1/doc/search-index.js"

if ! [ -e $index ]
then
    echo "Could not find the search index (looked for $index)"
    exit 1
fi

ins=$(grep -o 'In: .*' $source | sed 's/In: \(.*\)/\1/g')
outs=$(grep -o 'Out: .*' $source | sed 's/Out: \(.*\)/\1/g')

for p in $ins
do
    if ! grep -q $p $index
    then
        echo "'$p' was erroneously excluded from search index."
        exit 1
    fi
done

for p in $outs
do
    if grep -q $p $index
    then
        echo "'$p' was erroneously included in search index."
        exit 1
    fi
done

exit 0
