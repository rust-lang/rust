include ../tools.mk

all:
	echo a | $(CGREP) a
	! echo b | $(CGREP) a
	echo xyz | $(CGREP) x y z
	! echo abc | $(CGREP) b c d
	printf "x\ny\nz" | $(CGREP) x y z

	echo AbCd | $(CGREP) -i a b C D
	! echo AbCd | $(CGREP) a b C D

	true | $(CGREP) -v nothing
	! echo nothing | $(CGREP) -v nothing
	! echo xyz | $(CGREP) -v w x y
	! echo xyz | $(CGREP) -v x y z
	echo xyz | $(CGREP) -v a b c

	! echo 'foo bar baz' | $(CGREP) 'foo baz'
	echo 'foo bar baz' | $(CGREP) foo baz
	echo 'x a `b` c y z' | $(CGREP) 'a `b` c'

	echo baaac | $(CGREP) -e 'ba*c'
	echo bc | $(CGREP) -e 'ba*c'
	! echo aaac | $(CGREP) -e 'ba*c'

	echo aaa | $(CGREP) -e 'a+'
	! echo bbb | $(CGREP) -e 'a+'

	echo abc | $(CGREP) -e 'a|e|i|o|u'
	! echo fgh | $(CGREP) -e 'a|e|i|o|u'
	echo abc | $(CGREP) -e '[aeiou]'
	! echo fgh | $(CGREP) -e '[aeiou]'
	! echo abc | $(CGREP) -e '[^aeiou]{3}'
	echo fgh | $(CGREP) -e '[^aeiou]{3}'
	echo ab cd ef gh | $(CGREP) -e '\bcd\b'
	! echo abcdefgh | $(CGREP) -e '\bcd\b'
	echo xyz | $(CGREP) -e '...'
	! echo xy | $(CGREP) -e '...'
	! echo xyz | $(CGREP) -e '\.\.\.'
	echo ... | $(CGREP) -e '\.\.\.'

	echo foo bar baz | $(CGREP) -e 'foo.*baz'
	! echo foo bar baz | $(CGREP) -ve 'foo.*baz'
	! echo foo bar baz | $(CGREP) -e 'baz.*foo'
	echo foo bar baz | $(CGREP) -ve 'baz.*foo'
