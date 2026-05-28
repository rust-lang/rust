// exact-check
// ignore-tidy-linelength
const EXPECTED = [
    // should match (substring)
    {
        'query': 'struct:now::Country',
        'others': [
            { 'path': 'x::now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their', 'name': 'Country' },
        ],
    },
    {
        'query': 'struct:is::Country',
        'others': [
            { 'path': 'x::now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their', 'name': 'Country' },
        ],
    },
    {
        'query': 'struct:is_the::Country',
        'others': [
            { 'path': 'x::now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their', 'name': 'Country' },
        ],
    },
    {
        'query': 'struct:the::Country',
        'others': [
            { 'path': 'x::now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their', 'name': 'Country' },
        ],
    },
    {
        'query': 'struct:their::Country',
        'others': [
            { 'path': 'x::now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their', 'name': 'Country' },
        ],
    },
    // should not match
    {
        'query': 'struct:ood::Country',
        'others': [],
    },
    {
        'query': 'struct:goo::Country',
        'others': [],
    },
    {
        'query': 'struct:he::Country',
        'others': [],
    },
    {
        'query': 'struct:heir::Country',
        'others': [],
    },
    {
        'query': 'struct:hei::Country',
        'others': [],
    },
    {
        'query': 'struct:no::Country',
        'others': [],
    },
    // should match (edit distance)
    {
        'query': 'struct:nowisthetimeforallgoodmentocometotheaidoftheir::Country',
        'others': [
            { 'path': 'x::nowisthetimeforallgoodmentocometotheaidoftheir', 'name': 'Country' },
            { 'path': 'x::now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their', 'name': 'Country' },
        ],
    },
    {
        'query': 'struct:now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their::Country',
        'others': [
            { 'path': 'x::now_is_the_time_for_all_good_men_to_come_to_the_aid_of_their', 'name': 'Country' },
            { 'path': 'x::nowisthetimeforallgoodmentocometotheaidoftheir', 'name': 'Country' },
        ],
    },
];
