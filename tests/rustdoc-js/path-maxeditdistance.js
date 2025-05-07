// exact-check

const EXPECTED = [
    {
        'query': 'xxxxxxxxxxx::hocuspocusprestidigitation',
        // do not match abracadabra::hocuspocusprestidigitation
        'others': [],
    },
    {
        // exact match
        'query': 'abracadabra::hocuspocusprestidigitation',
        'others': [
            { 'path': 'abracadabra', 'name': 'HocusPocusPrestidigitation' },
        ],
    },
    {
        // swap br/rb; that's edit distance 1, where maxPathEditDistance = 2
        'query': 'arbacadarba::hocuspocusprestidigitation',
        'others': [
            { 'path': 'abracadabra', 'name': 'HocusPocusPrestidigitation' },
        ],
    },
    {
        // swap p/o o/p, that's also edit distance 1
        'query': 'abracadabra::hocusopcusprestidigitation',
        'others': [
            { 'path': 'abracadabra', 'name': 'HocusPocusPrestidigitation' },
        ],
    },
    {
        // swap p/o o/p and gi/ig, that's edit distance 2
        'query': 'abracadabra::hocusopcusprestidiigtation',
        'others': [
            { 'path': 'abracadabra', 'name': 'HocusPocusPrestidigitation' },
        ],
    },
    {
        // swap p/o o/p, gi/ig, and ti/it, that's edit distance 3 and not shown (we stop at 2)
        'query': 'abracadabra::hocusopcusprestidiigtaiton',
        'others': [],
    },
    {
        // truncate 5 chars, where maxEditDistance = 2
        'query': 'abracadarba::hocusprestidigitation',
        'others': [],
    },
    {
        // truncate 9 chars, where maxEditDistance = 2
        'query': 'abracadarba::hprestidigitation',
        'others': [],
    },
];
